#venv Creation: 
1. python -m venv env_name // "C:/Users/cryin/AppData/Local/Programs/Python/Python312/python.exe" -m venv modelVenv
2. cd Scripts -> activate.bat
3. pip list #to list packages installed
4. to create requirements.txt : pip freeze > requirements.txt
5. to install from requirements.txt : pip install -r requirements.txt
6. Deactivate : deactivate

#FLASK server

the app.py is a flask server, running it starts the server that takes the API /predict to take image

using curl to test instead of thunderclient:
curl -X POST -F "image=@C:\Users\cryin\Work\vc-emotion-detection-master\dataset\train\disgusted\im13.png" http://localhost:5000/predict


#camera.py
It is a module to start camera and directly show the emotions there instead of going through a GUI or frontend...
hieeeee