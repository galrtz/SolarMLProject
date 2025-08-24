
DATA EXPLANTION:

Each CSV file represents a single time step, which is subsequently transformed into a graph structure. 
In this representation, nodes correspond to PV sites and their associated features, while edges capture the spatial
relationships among them. The collection of CSV files thus forms a temporal sequence of graphs, suitable for spatio-temporal
learning tasks.

The test dataset is derived from 2017, covering a continuous two-week period from January 5th to January 19th. This portion is
held out exclusively for evaluation, ensuring an unbiased measurement of the model’s generalization capability.

The training dataset is constructed from 2018 records, after conversion into graphs. Feature selection was carefully designed:
instead of using all timestamps, only hourly rounded values were preserved, incorporating both short-term dependencies(the
previous two hours) and long-term historical references (24, 48, and 72 hours back). This combination provides a multi-scale
temporal context without introducing unnecessary redundancy.

To further prevent temporal leakage and enhance robustness, the training subset was restricted to every fifth day only, rather than
consecutive days. Within each sampled day, data points were sampled at two-hour intervals instead of continuously throughout
the day. This sampling strategy enforces temporal decorrelation, increases variability, and prevents the model from overfitting 
to densely correlated sequences.

