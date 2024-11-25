import huggingface_hub
import os

token=os.environ["APITOKEN"]
huggingface_hub.login(token=token)



