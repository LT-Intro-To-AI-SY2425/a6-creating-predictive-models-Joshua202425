# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
The model is not very accurate, 0.68, because the data is no lnger scaled which causes there to be outliers that skew the data.
2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
The model is very accurate, 0.92, with the scaler to the point where it is accurate enough for the given use case due to it showing high correlation.
3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
The model did exceptionally well; however, there is a pattern that caused the model to get a few predictions wrong which was just the repitition of one or zero many times in a row.
4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
No she would not.
