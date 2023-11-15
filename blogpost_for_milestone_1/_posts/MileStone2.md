---
layout: post
title: Milestone 2
---

## **Task 2 - Feature Engineering I**

### Question 1

After preparing the tidy dataset to include details about all shots and goals, we introduced several new attributes, including "shot distance," "shot angle," "goal status," and "empty net." With this additional information, we were able to create the following visualizations:

**Shot counts by distance**

![shot counts by distance](/assets/images/shot_by_distance.PNG)

As depicted in the figure above, it is clear that the majority of shots and goals were attempted from distances ranging between 5 to 70 feet, which aligns with our expectations. The greatest concentration of both shots and goals occurred at a distance of approximately 15 feet. In contrast, very few shots were taken from the defensive side of each team, where the distance exceeded 100 feet.

**Shot counts by angle**

![shot counts by angle](/assets/images/shot_by_angle.PNG)

The shot count histogram, categorized by shot angle, reveals that hockey players tend to favor shots from the front (with angles ranging from 0 to 45 degrees) over shots from the sides with wider angles. This preference is reasonable, as taking a shot with a very wide angle increases the likelihood of missing the target.

It's worth noting that there were instances where shot angles exceeded 90 degrees. This indicates that hockey players attempted shots from behind the net, which is quite remarkable!

**Shot by both distance and angle from the net**

![shot distance and angle](/assets/images/shot_by_distance_and_angle.PNG)

By considering both shot distance and shot angle together, we can make the following observations:

- Shots with angles exceeding 90 degrees are consistently taken close to the net. This is evident because the maximum distance behind the net is approximately 25 feet. This finding reinforces the appropriateness of our feature engineering process.

- Shots taken from a significant distance are consistently launched at angles ranging from 0 to 25 degrees. This aligns with expectations, as taking a shot very close to the net with a steep angle would greatly increase the chances of the shot hitting the goal post.

### Question 2

In this section, we explore the relationship between the goal rate and two key features: shot distance and shot angle:

![goal rate by distance](/assets/images/goal_rate_by_distance.PNG)

![goal rate by angle](/assets/images/goal_rate_by_angle.PNG)

- The first chart illustrates that as shots get closer to the target, their likelihood of becoming a goal increases. However, there's an interesting observation where shots from distances greater than 160 feet exhibit a notably higher goal rate compared to those in the 40 to 80 feet range. This might be due to the limited number of shots taken from these extremely long distances, making the goal rate for far-distance shots less reliable.

- The second chart also highlights the challenges of using goal rate as an evaluation metric for hockey shots. For instance, there is only one shot with a 100-degree angle, and it happened to be a goal. This results in a 100% goal rate for the 100-degree shots, which is not a sensible representation. In my opinion, one way to address this issue is to weight the goal rate by a coefficient that is inversely proportional to the shot counts, making the "goal rate" feature more informative and robust.

### Question 3

In this part, we attempt to identify potential errors within the dataset. In particular, we have an intuitive understanding that **it is an extremely rare occurrence to score a non-empty net goal against the opposing team from your defensive zone**. To verify our expectations, we have generated figure depicting shots taken with an empty net and shots taken without an empty net. 

Our intention is to scrutinize the presence of far-distance shots taken without an empty net and investigate the underlying reasons if such shots are observed.

![goal non empty net](/assets/images/goal_non_empty_net_by_distance.PNG)

The figure above clearly indicates the existence of shots within the range we described (between 150 to 170 feet) when the net was not empty.

Upon examining the processed dataset, we have identified incorrect or inaccurate data.

![wrong coordinates dataframe](/assets/images/wrong_coordinates.png)

As an example, consider the shot taken at 10:27 during the second period of the game between FLA and DET on December 24, 2016, which was recorded with incorrect coordinates and can be seen in the above figure (highlighted in white).

Upon closer inspection of the dataframe, it is apparent that the recorded shot coordinates are [-97, 21], while the side of the net being shot at is on the right, which should correspond to coordinates [89, 0]. This erroneous recording has led to an inaccurate calculation of the shot distance (187 feet), whereas the actual distance is very close to the net. In reality, the correct coordinates for this shot were [97, -21].

Further investigation into similar shots has confirmed that the recorded coordinates are incorrect for all of these cases.

## **Task 4 - Feature Engineering II**

In this undertaking, we have modified the organized tidy data to incorporate additional essential features with the anticipation of enhancing the performance of machine learning algorithms.

To be precise, the supplementary features are detailed in the table below:

|Column Name   |Description and Explanation   |
|-------|--------------------------------------|
|game_seconds|{::nomarkdown}The timestamp indicating when the shot was executed in seconds{:/}|
|period|{::nomarkdown}The period during which the shot was taken{:/}|
|x_coordinate|{::nomarkdown}x-coordinate of the shot{:/}|
|y_coordinate|{::nomarkdown}y-coordinate of the shot{:/}|
|shot_distance|{::nomarkdown}The shot's distance from the net goal{:/}|
|shot_angle|{::nomarkdown}The degree measure of the angle between the net and the shot{:/}|
|emptyNet|{::nomarkdown}A binary variable that denotes whether the net was unoccupied when the shot was executed{:/}|
|x_last_event|{::nomarkdown}x-coordinate of the previous event{:/}|
|y_last_event|{::nomarkdown}y-coordinate of the previous event{:/}|
|time_from_last_event|{::nomarkdown}The duration of time between the present shot and the preceding event{:/}|
|distance_from_last_event|{::nomarkdown}The measurement, in feet, of the separation between the current shot and the coordinates of the prior event{:/}|
|is_rebound|{::nomarkdown}A boolean variable indicating whether the shot is a rebound{:/}|
|change_in_shot_angle|{::nomarkdown} The difference in angle between two successive shots, measured in degrees. This calculation applies only if the preceding event was a shot or a goal; otherwise, it remains at 0{:/}|
|speed|{::nomarkdown} Calculated as the ratio of the distance from the prior event to the time elapsed since the prior event{:/}|
|time_power_play|{::nomarkdown}A timer that records the duration of the game in a power-play situation{:/}|
|num_player_home|{::nomarkdown}The count of players from the home team present when the shot was executed{:/}|
|num_player_away|{::nomarkdown}The count of players from the away team present when the shot was taken{:/}|

It's important to note that we have already rectified the erroneous coordinate information to obtain this final dataframe.

Regarding rebound shots, we have included shots where the preceding event was either a shot or a **blocked shot**, which deviates slightly from the task's initial suggestion.

Furthermore, we identified numerous instances in which two consecutive shots were separated by a significant time interval, such as 22 seconds. In such cases, we do not consider them as rebounds. Therefore, we define a shot as a rebound only if the time gap between it and the previous shot is less than 4 seconds.

We are supplying a dataframe that encompasses all the game-related information for the matchup between Winnipeg and Washington that took place on March 12, 2018, with a game ID of *2017021065*.

The [dataframe can be accessed here](https://www.comet.com/2nd-milestone/feature-engineering-data/2d7198d091ec403a85e33d3b9cadce18?assetId=16bc437b6fa34861a70713e411b1101a&assetPath=dataframes&experiment-tab=assetStorage)
