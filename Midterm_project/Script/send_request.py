import requests
data = {
    "weather_temp": 16.74,      
    "weather_feels_like":     16.72 ,     
    "weather_humidity":       86.00    ,    
    "weather_pressure":     1013.00   ,   
    "weather_wind_speed" :     0.38    ,   
    "weather_wind_deg"   :    42.00     , 
    "weather_wind_gust"   :    0.80    ,    
    "weather_clouds"     :    49.00   ,
    "weather_visibility": 10000.00
}
url = "http://host.docker.internal:9000/predict"  
print(requests.post(url, json=data).json())

