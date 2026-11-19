# Clinic no-show tree

Should the clinic overbook a slot?

`python3 analysis.py` refits the tree and rewrites `slides.pdf`. Every figure on the slides comes from that run.

- Data: `clinic_appointments.csv`, 400 past appointments.
- Split: stratified 70/30, `random_state=42` (280 / 120).
- Model: classification tree, depth at most 3, at least 10 training rows in a leaf, minimum impurity decrease 0.01.
- Decision: overbook when the tree predicts a no-show.
- Yearly value assumes 20 appointments a day, 5 days a week, 50 weeks.
