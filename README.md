# spotify_danceability
- The development of a Machine Learning model that guesses the danceability scores of Spotify songs ( (of 140k tracks in Spotify). 
- The model uses gradient boosting and hyperparameter optimization to achieve the most accurate results -less than 10% error!
- The project is able to create the playlist on user's own account using the Spotify Python API!
- model.py - Inludes whole model structure. At the end of hyperparameter optimization on 3 different models we got an RMSE score less than 0.1.
- model_ML_Pipeline.py : When you run this on python from cmd, a webapp starts locally on your machine to run the program.
- requirements.txt: Python module dependencies and versions are here.

### Installation:
  pip install -r requirements.txt (works better on pycharm virtual environment!)

### Creation of dummy Spotify project (Will be done only once!):
  TThe link explaining it: https://www.codeproject.com/Tips/5276627/HowTo-Setup-a-Spotify-API-App-in-the-Spotify-Devel
  
### How to run (on Windows):
  1. Install dependencies as written below in "Installation"
     pip install -r requirements.txt on command line \n
  2. Run "model_ML_Pipeline.py" from commandline \n
  3. On command line there is a local address that web app tuns on your machine, click on it to open app. \
  5. Enter your spotify userid, client id (from dummy Spotify project) and client secret (from dummy Spotify project) to web app as inputs and hit 'Submit' \n
  6. To get your spotify user id: \n
	From "https://open.spotify.com/" ron top right click on "Account". \n
	On the new tab  click "Edit profile" and there you have your user id! \n
	 
  7. Web app runs and gives you the output as "Hello 12345678912 The ML Dance Playlist has been created:  \n
	ML Dance Playlist 14_48_time". \n
