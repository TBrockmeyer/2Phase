"# 2Phase" 
# Script reads an image of chemical vials, detects location of transition between water- and oil-based phase

# 1) reads an image of (usually 8) vials with chemical substances
# 2) optionally reduces the colors to, e.g., 32 colours
# 3) cuts segments out of the pictures: for every vial, one segment, size e.g.: h=805 px, w=100 px
# For each segment, the following operations are done:
# 4) resize to height of h (usually 100) px
# 5) depending on how many colors are detected, the threshold (~coarse or fine) is calculated with which the colored areas are distinguished from each other: if a vial is almost colorless, detection will have to be finer
# 6) areas of transition between colored areas are identified. In between, a line (position) is previsioned
# 7) if two lines have less than d pixels (e.g., 0.15*segment height) in between, a line between these two will replace them
# 8) if the are above and below one line are similar in colour, the line will be deleted
