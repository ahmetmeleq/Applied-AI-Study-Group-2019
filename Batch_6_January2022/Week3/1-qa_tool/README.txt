This readme is for linux users.
If you use windows, some steps might be run with different commands.

1- create new environment
conda create -n qatool

2- activate the env
conda activate qatool

3- install pip to environment
conda install pip

4- check if "pip" keyword invokes the right pip program.
it should be in "../envs/qatool/bin/pip"
which pip

5- use pip to install requirements
pip install -r applied_qatool.txt

6- assign FLASK_APP variable for flask
export FLASK_APP=app.py

7- run your flask application
flask run

8- go to link: http://127.0.0.1:5000/ !