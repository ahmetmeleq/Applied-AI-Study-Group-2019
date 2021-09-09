from flask import Flask
myapp=Flask(__name__,template_folder='templates',static_folder='static:css')
from myapp import main
