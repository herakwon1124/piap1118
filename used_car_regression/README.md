# Used car price regression

Should the dealer buy a car at auction?

`python3 analysis.py` refits the regression and rewrites `slides.pdf`. Every figure on the slides comes from that run.

- Sales data: `used_car_sales.csv`, 240 cars sold in the last year.
- Auction data: `auction_lots.csv`, 40 cars. Not used to fit the model.
- Split: 70% train / 30% test, `random_state=42` (168 / 72).
- v1: sale price on mileage, age, horsepower, and accident. Number of previous owners is dropped.
- Decision: buy if the predicted sale price exceeds the auction price plus $1,500.
