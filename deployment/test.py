#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

patient = {
   'gender': 'male',
   'age': 51.0,
   'hypertension': 'present',
   'heart_disease': 'present',
   'ever_married': 'yes',
   'work_type': 'private',
   'residence_type': 'rural',
   'avg_glucose_level': 166.29,
   'bmi': 25.6,
   'smoking_status': 'formerly_smoked'
}

response = requests.post(url, json=patient).json()
print(response)