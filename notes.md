### Completed:
**Models implemented:**  
- RF
- FDA
- GAM (`pygam`)
- CTA
- RF
- GLM (~~`statsmodels`~~ `scikit pipeline`)
- MARS (`py-earth`)
- GBM
- MaxEnt (`elapid`?)

Embed R scripts, vary the number of absences generated - this mightve actually been detrimental to accuracy

Resolve ocean presence generation

Implement thresholds with max sss

Find a method to choose models for ensemble modelling - auc and tss

Assign classifiers to each type of absence generation

Ensemble modelling - how do you choose 3 models?

Display data better 

Sampling bias?

myabe also thin the presence sightings within ~~10km~~ 5 km of each other?

check pearsons correlation coeeff

Investigating field studies of NST efficacy in areas with known infestations (sorta done? will end up being done in discussion)

check feature weight (i only did for tree and linear models)

Make gif setup of the spreading range over the course of the next few decades 

clean up the month range of each pest based on crop

current bio vars are not that cooked, but try the new ones (Including soil/more bioclimactic variables? out of the backburner!)

I really don't like modifying the R script to include nonexistent uncertainty.... but that is the only way seedcorn maggot sightings will work...

Solution: run all of them, with seedcorn maggots being the only exception

### To Do:

develop into a usable tool? where you dont have to understand the entire codebase in order to use it


### Backburner
Possibly generate pseudo absence points using environmentally stratified methods   
   
continuous boyce index and how that might be helpful 

spatial blocking cross validation, possibly through verde scikit - this may reduce sampling bias



