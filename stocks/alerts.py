# # alerts.py
# from twilio.rest import Client
# from django.conf import settings

# def send_sms_alert(message, to_phone_number):
#     client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
#     client.messages.create(
#         body=message,
#         from_=settings.TWILIO_PHONE_NUMBER,
#         to=to_phone_number
    # )
    
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

def trigger_alerts(stock, action, price, quantity):
    subject = f"Trading Alert for {stock}"
    message = f"The bot executed a {action} for {quantity} shares of {stock} at ${price}."
    recipient_list = ['user@example.com']
    
    # Send email alert
    send_email_alert(subject, message, recipient_list)
    
    # Send SMS alert
    send_sms_alert(message, '+1234567890')
    
    # Send browser notification
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        'alerts',
        {
            'type': 'send_alert',
            'message': message
        }
    )
