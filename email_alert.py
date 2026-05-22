import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import datetime

# Your email details
SENDER_EMAIL = "keerthanasivakumar084@gmail.com"
APP_PASSWORD = "osdpienjhpcyanld"
RECEIVER_EMAIL = "keerthanasivakumar084@gmail.com"

def send_alert(threat_score, screenshot_path=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = f"🚨 CRIMEVIS AI - THREAT DETECTED! Score: {threat_score}/10"

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
CRIMEVIS AI ALERT

Time: {now}
Threat Score: {threat_score}/10
Status: IMMEDIATE ATTENTION REQUIRED

This is an automated alert from CRIMEVIS AI.
        """
        msg.attach(MIMEText(body, 'plain'))

        # Attach screenshot if available
        if screenshot_path:
            with open(screenshot_path, 'rb') as f:
                img = MIMEImage(f.read())
                msg.attach(img)

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"ALERT EMAIL SENT! Threat Score: {threat_score}/10")
        return True

    except Exception as e:
        print(f"Email error: {e}")
        return False
