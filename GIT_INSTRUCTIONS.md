If you working on your own machine or are having issues syncing the workshop git repository, please consult these suggested directions. First, clone this git repository if it doesn't exist

        git clone git@github.com:NCAR/GPU_workshop.git

Prior to each session, we recommend that you update to the latest version of the teaching material repository by running the below commands. This will modify your working directoy files to match those listed on Github. If you have editted files which you don't care about, 

        # If you have editted files you don't want to save, first reset your repo (will permanently overwrite files to match Github commit)
        git reset --hard
        
        # Otherwise, checkout workshop branch and update it to latest version
        git checkout workshop
        git pull git@github.com:NCAR/GPU_workshop.git

When working on course material individually, we recommend you create your own local branch for that session so that your personal edits won't be deleted as course contents may update throughout the workshop series. After checking out and updating the workshop branch, run the below to create and switch to your own branch. If desired, you should then save and commit any changes you make to your own branch before returning to the workshop branch and pulling recent changes with the prior commands.

        git checkout -b session##
        # Do work
        git add [file_to_save]
        git commit -m "Session## WIP"

If at a future time you want to revisit any personal work done in a specific session, simply checkout your local branch with the below command.

        git checkout session##

We will do our best to minimize changes to each session's course materials once published but nonetheless, management of any merge conflicts will be left to the user. Lastly, we encourage regular improvement and contributions to this course material. Feel free to submit pull requests via a forked repository or reach out to workshop organizers with your feedback.