# stocks/consumers.py
# import json
# from channels.generic.websocket import AsyncWebsocketConsumer

# class AlertConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()

#     async def disconnect(self, close_code):
#         pass

#     async def receive(self, text_data):
#         data = json.loads(text_data)
#         message = data['message']

#         await self.send(text_data=json.dumps({
#             'message': message
#         }))

# stocks/consumers.py

import json
from channels.generic.websocket import AsyncWebsocketConsumer

# stocks/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class AlertConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        # Send a test notification
        await self.send(text_data=json.dumps({
            'message': 'Test notification'
        }))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data['message']

        await self.send(text_data=json.dumps({
            'message': message
        }))

# class AlertConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.channel_layer.group_add("alerts", self.channel_name)
#         await self.accept()

#     async def disconnect(self, close_code):
#         await self.channel_layer.group_discard("alerts", self.channel_name)

#     async def receive(self, text_data):
#         await self.send(text_data=json.dumps({"message": text_data}))

#     async def send_alert(self, event):
#         message = event['message']
#         await self.send(text_data=json.dumps({"message": message}))

# import json
# from channels.generic.websocket import AsyncWebsocketConsumer

# class AlertConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()
#         # Send a test notification
#         await self.send(text_data=json.dumps({
#             'message': 'Test notification'
#         }))

#     async def disconnect(self, close_code):
#         pass

#     async def receive(self, text_data):
#         data = json.loads(text_data)
#         message = data['message']

#         await self.send(text_data=json.dumps({
#             'message': message
#         }))