# Register VQA as a service in Qanary
import requests
import json
import time
from app import app

def getHTTPHeaders():
    return { 
             'Content-Type': 'application/json',
             'Accept': 'application/json'
            }

def compareAndSetRegisteredId(response_id):
    # TODO store the id somewhere
    return True

def register():
    service_url = "http://localhost:8080"
    headers = getHTTPHeaders()
    response = requests.post(service_url, headers=headers)    

    if response.status_code == 200:
        response_body = json.loads(response.text)
        if compareAndSetRegisteredId(response_body["id"]):
            print "Application registered as %s"%response_body
        else:
            print "Application refreshed itself as %s"%response_body
        return True
    else:
        print "Application failed to register as Visual Question Answering. Response:%s"%response.text
        return False


if __name__ == "__main__":
    while True:
        if not register():
            break
        time.sleep(10)