# Testing the Compute Environment on Casper and Other Systems
Please follow the directions below and in the notebook [TestCasper.ipynb](TestCasper.ipynb) to run some basic GPU program tests and make sure you're able to access NCAR's compute clusters and your user account has the correct access permissions to run relevant workshop compute jobs under the course's project ID **UCIS0004**. Registered workshop participants should have received details about their NCAR CIT account if a new user and established users should have had the prior project ID added to their account and now listed at [sam.ucar.edu](https://sam.ucar.edu).

To initialize and run this notebook on Casper: 
1. Start a Jupyter Hub session via the **[NCAR JupyterHub portal](https://jupyterhub.hpc.ucar.edu/stable/)**
2. For this notebook, choose "**Casper login node**" under the "Cluster Selection" pulldown.
3. **Navigate to the folder** you'd prefer to save the GPU workshop github repo (Default is your `$HOME` folder)
4. Select the **git icon** (diamond square below Dask icon) on the side panel at the left side of the browser window
5. Select "**Clone a Repository**" 
6. Enter this git repository address **`https://github.com/dphow/GPU_workshop.git`**
7. Navigate into the newly cloned `GPU_workshop` directory and select the file **`TestCasper.ipynb`** in the folder `00_TestCasper`

Once you have this notebook displayed and running under the Bash kernel (check active kernel in top right of window), then run each code cell below in order by selecting the cell and pressing CTRL+ENTER.

If you have any questions, problems running this notebook, or issues accessing the compute cluster, please reach out to workshop organizers over email or the [NCAR GPU Users Slack](https://ncargpuusers.slack.com).