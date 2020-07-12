#!/usr/bin/python
from flask import Flask, render_template, request
from flask_restx import Api, Resource, fields
import joblib
import pickle
from model_deployment import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='''Based on movie year, title and plot, this API predicts the genres of the movie
        Universidad de Los Andes
        Master in Analytic Intelligence for Decision Making
        Advanced Methods in Data Analysis
        
        Carlos Francisco Silva Ortiz - 201920463
        Diana Rocío Díaz Rodríguez - 201331684
        Javier Alfonso Lesmes Patiño - 200820243''')

ns = api.namespace('Movie Genres Prediction', 
     description='Predicts the genres of the movie')
   
parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Movie Year', 
    location='args')
    
parser.add_argument(
    'Title', 
    type=str, 
    required=True, 
    help='Title of the movie', 
    location='args')

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Plot fo the movie', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PriceForecast (Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['Year'],args['Title'],args['Plot'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
