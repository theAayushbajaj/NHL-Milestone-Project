from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    '''
    This function reads in the raw data from the 2016 season and creates a tidy dataset
    that can be used for analysis. The tidy dataset is saved as a csv file in the data folder.

    Parameters
    ----------
    None

    '''
    # Create a SparkSession
    spark = (
        SparkSession.builder
        .config('spark.executor.memory', '3g')
        .config('spark.executor.cores', '4')
        .config('spark.driver.memory','5g')
        .config('spark.cores.max', '300')
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config('spark.sql.shuffle.partitions', 20)
        .config('spark.sql.debug.maxToStringFields','50')
        .getOrCreate()
        )
    logging.info("SparkSession created")
    # Read JSON files into a DataFrame
    df_for_schema = spark.read.json("data/data/raw_data/2016/regular_games/2016021230.json")
    logging.info("JSON file schema read into a DataFrame")

    # Create a schema for the data
    schema = df_for_schema.schema

    # Read all JSON files into a DataFrame
    json_compiled = spark.read.option("recursiveFileLookup", "true").json("data/data/raw_data/*/*/*.json", schema=schema)
    logging.info("All JSON files read into a DataFrame")

    # Select the columns of interest
    allplay = json_compiled.select(F.col("gamePk").alias('game_id'),
                           F.col("gameData.game.season").alias("season"),
                            F.col("gameData.game.type").alias("game_type"),
                            F.col("gameData.datetime.dateTime").alias("start_time"),
                            F.col("gameData.datetime.endDateTime").alias("end_time"),
                            F.col("gameData.teams.away.id").alias("away_team_id"),
                            F.col("gameData.teams.away.name").alias("away_team_name"),
                            F.col("gameData.teams.home.id").alias("home_team_id"),
                            F.col("gameData.teams.home.name").alias("home_team_name"),
                            F.explode(F.col("liveData.plays.allplays")).alias("allplays") # Explode the allplays column since it is a list
                                        )
    logging.info("Columns of interest selected")
    
    linescore = json_compiled.select(F.col("gamePk").alias('game_id'),
                                    F.explode(F.col("liveData.linescore.periods")).alias("periods")
                                    )
    linescore = linescore.select("game_id",
                            F.col("periods.num").alias("period"),
                            F.col("periods.home.rinkSide").alias("home_rinkSide"),
                            F.col("periods.away.rinkSide").alias("away_rinkSide"),
            )
    logging.info("Linescore columns of interest selected")
    # Filter for only goals and shots
    allplay = allplay.filter((allplay.allplays.result.event=="Goal") | (allplay.allplays.result.event=="Shot"))
    logging.info("Filtered for only goals and shots")
    # Select the columns of interest
    allplayDF = allplay.select(F.col("game_id"),
                           F.col("season").alias("season"),
                            F.col("game_type").alias("game_type"),
                            F.col("start_time").alias("start_time"),
                            F.col("end_time").alias("end_time"),
                            F.col("away_team_id").alias("away_team_id"),
                            F.col("away_team_name").alias("away_team_name"),
                            F.col("home_team_id").alias("home_team_id"),
                            F.col("home_team_name").alias("home_team_name"),
                            F.col("allplays.about.eventIdx").alias("eventIdx"),
                           F.col("allplays.result.event").alias("event"),
                           F.col("allplays.result.description").alias("description"),
                           F.col("allplays.result.secondaryType").alias("shotType"),
                           F.col("allplays.result.strength.code").alias("strength"),
                           F.col("allplays.result.gameWinningGoal").alias("gameWinningGoal"),
                           F.col("allplays.result.emptyNet").alias("emptyNet"),
                            F.col("allplays.result.penaltySeverity").alias("penaltySeverity"),
                            F.col("allplays.result.penaltyMinutes").alias("penaltyMinutes"),
                            F.col("allplays.coordinates.x").alias("x_coordinate"),
                            F.col("allplays.coordinates.y").alias("y_coordinate"),
                            F.col("allplays.team.name").alias("team_name"),
                            F.col("allplays.about.period").alias("period"),
                            F.col("allplays.about.periodType").alias("periodType"),
                            F.col("allplays.about.periodTime").alias("periodTime"),
                            F.col("allplays.about.periodTimeRemaining").alias("periodTimeRemaining"),
                            F.when(F.col("allplays.result.event")=="Goal",1).otherwise(0).alias("is_goal"),
                            F.explode(F.col("allplays.players")).alias("players") # Explode the players column since it is a list of Goalie, Shooter, Assist, and Scorer
                            )
    logging.info("Columns of interest in livedata configured and selected")
    
    # Groupby game_id and eventIdx and pivot on playerType to get the player names
    player_characteristics = allplayDF.groupBy(["game_id","eventIdx"]).pivot("players.playerType").agg(F.first("players.player.fullName"),F.last("players.player.fullName"))
    logging.info("Player characteristics pivot table aggregated")
    # Rename the columns to be more descriptive
    player_characteristics = player_characteristics.withColumnRenamed("Assist_first(players.player.fullName)","Assist_first")\
            .withColumnRenamed("Assist_last(players.player.fullName)","Assist_last")\
                .withColumnRenamed("Scorer_first(players.player.fullName)","Scorer")\
                    .withColumnRenamed("Goalie_first(players.player.fullName)","Goalie")\
                        .withColumnRenamed("Shooter_first(players.player.fullName)","Shooter")
    logging.info("Player characteristics columns renamed")
    player_characteristics = player_characteristics.select("game_id","eventIdx","Assist_first","Assist_last","Scorer","Goalie","Shooter")
    
    # Join the player_characteristics to the allplayDF
    allplayDF = allplayDF.join(player_characteristics,["game_id","eventIdx"],"left").drop("players").drop_duplicates()
    allplayDF = allplayDF.join(linescore,["game_id","period"],"left").drop_duplicates()
    logging.info("Player characteristics and linescore joined to allplayDF")
    
    # Calculate the distance to the goal
    allplayDF = allplayDF.withColumn("rinkSide",F.when(F.col("team_name")==F.col("home_team_name"),F.col("home_rinkSide")).otherwise(F.col("away_rinkSide")))
    allplayDF = allplayDF.withColumn("x_goal",F.when(F.col("rinkSide")=="left",89).otherwise(-89))
    allplayDF = allplayDF.withColumn("y_goal",F.lit(0))
    allplayDF = allplayDF.withColumn("shot_distance",F.sqrt((F.col("x_coordinate")-F.col("x_goal"))**2 + (F.col("y_coordinate")-F.col("y_goal"))**2))
    logging.info("Distance to goal calculated")
    
    # Write the tidy dataset to a csv file 
    allplayDF.write\
        .option('header', 'true')\
        .format('com.databricks.spark.csv')\
        .mode('overwrite').csv('data/playData.csv',compression="gzip", sep=",")
    logging.info("Tidy dataset written to csv file")

if __name__ == "__main__":
    main()