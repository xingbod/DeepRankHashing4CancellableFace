import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import argparse
import sklearn
from scipy.signal import savgol_filter
######################################################################
### Parameter and arguments

parser = argparse.ArgumentParser(description='Evaluate unlinkability for two given sets of mated and non-mated linkage scores.')
parser.add_argument('matedScoresFile', help='filename for the mated scores', type=str)
parser.add_argument('nonMatedScoresFile', help='filename for the non-mated scores', type=str)
parser.add_argument('figureFile', help='filename for the output figure', type=str)
parser.add_argument('--omega', help='omega value for the computations, if none provided, omega = 1', nargs='?', default=1., type=float)
parser.add_argument('--nBins', help='number of bins for the computations, if none provided, nBins = min(length(matedScoresFile) / 10, 100)', nargs='?', default=-1, type=int)
parser.add_argument('--figureTitle', help='title for the output figure', nargs='?', default='Hamming Similarity Analysis', type=str)
parser.add_argument('--legendLocation', help='legend location', nargs='?', default='upper right', type=str)


# e.g.  python evaluateRevocability.py ../matlab/psedo_mated.txt ../matlab/psedo_non_mated.txt ../matlab/mated.txt  revocability_nbins50.svg --nBins 50
# e.g.  python evaluateRevocability.py ../matlab/permlut_imposter_mated.txt ../matlab/permlut_imposter.txt ../matlab/permlut_genuine.txt  revocability_permlut_nbins50.svg --nBins 50

args = parser.parse_args()
matedScoresFile = args.matedScoresFile
nonMatedScoresFile = args.nonMatedScoresFile
figureFile = args.figureFile
figureTitle = args.figureTitle
legendLocation = args.legendLocation
omega = args.omega
nBins = args.nBins


######################################################################
### Evaluation

# load scores
# matedScores = numpy.fromfile(matedScoresFile)
# nonMatedScores = numpy.fromfile(nonMatedScoresFile)

matedScores = numpy.loadtxt(matedScoresFile)
nonMatedScores = numpy.loadtxt(nonMatedScoresFile)

# remove value 1 coz they are from same samples
matedScores = matedScores[matedScores!=1]

if nBins == -1:
	nBins = min(len(matedScores)/10,100)
#
# matedScores = savgol_filter(matedScores, 51, 3) # window size 51, polynomial order 3
# nonMatedScores = savgol_filter(nonMatedScores, 51, 3) # window size 51, polynomial order 3

# define range of scores to compute D
bin_edges = numpy.linspace(min([min(matedScores), min(nonMatedScores)]), max([max(matedScores), max(nonMatedScores)]), num=nBins + 1, endpoint=True)
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2 # find bin centers

# compute score distributions (normalised histogram)
y1 = numpy.histogram(matedScores, bins = bin_edges, density = True)[0]
y2 = numpy.histogram(nonMatedScores, bins = bin_edges, density = True)[0]


### Plot final figure of D + score distributions

plt.clf()

sns.set_context("paper",font_scale=1.7, rc={"lines.linewidth": 2.5})
sns.set_style("white")

ax = sns.kdeplot(matedScores, shade=False, label='Genuine', color=sns.xkcd_rgb["medium green"],linewidth=3)
x1,y1 = ax.get_lines()[0].get_data()
ax = sns.kdeplot(nonMatedScores, shade=False, label='Imposter', color=sns.xkcd_rgb["pale red"],linewidth=3, linestyle='--')
x2,y2 = ax.get_lines()[1].get_data()


# Figure formatting
ax.spines['top'].set_visible(False)
ax.set_ylabel("Probability Density")
ax.set_xlabel("Score")
ax.set_title("%s" % (figureTitle),  y = 1.02)

labs = [ax.get_lines()[0].get_label(), ax.get_lines()[1].get_label()]
lns = [ax.get_lines()[0], ax.get_lines()[1]]
ax.legend(lns, labs, loc = legendLocation)

ax.set_ylim([0, max(max(y1), max(y2)) * 1.05])
ax.set_xlim([bin_edges[0]*0.98, bin_edges[-1]*1.02])

plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(right=0.88)
pylab.savefig(figureFile)
