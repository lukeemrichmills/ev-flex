# ev-flex
V2X EV flexibility sim


# how I've done this

The simulator generates 2 days of charging and EV driving behaviour. I've taken the second day as the display simulator. This is so that the probabilities of charging and the SoC at the beginning of day 2 reflect a realistic lead-on from a previous day. This is akin to achieving a steady state before sampling in other simulators. 

I've taken an agent-based modelling approach, generating Communities of EVAgents and running different Experiments...

I added one extra field to the csv file to account for driving probability per day - this was to take into account the difference between Infrequent chargers and Infrequent drivers. This could probably use some work...

# notes on archetypes

I spent a bit too long figuring out Infrequent drivers - I settled on something that seems to work and represent the archetype on aggregate, but had to force the plug-in frequency per day. The miles per year (average) don't seem to square with the plug-in frequency but perhaps I'm missing something...

Also it didn't make sense to me that Always Plugged-in would have the average miles per year, so I set this to 0. This also made things easier to model.

The intelligent octopus average archetype essentially just represents a high EV user in this formulation, racking up almost 3x the amount of miles per year than the average EV driver.

# tests
