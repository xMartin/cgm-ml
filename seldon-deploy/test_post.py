import requests
import json
import numpy as np

addr = 'http://localhost:5000'
test_url = addr + "/cgm/regressor"

# Call server.
points = np.random.random(30000 * 4).astype("float32")
response = requests.post(test_url, data=points.tostring())#, headers=headers)
print(json.loads(response.text))
