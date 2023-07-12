from constants import IMAGES
from Grader import Grader

g = Grader(IMAGES)

# give an index as argument to save the corresponding student's status into a csv file
g.save_status(12)

g.save_all_status()
