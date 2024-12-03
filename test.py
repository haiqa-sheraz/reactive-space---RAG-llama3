import streamlit as st
from streamlit_chat import message
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
import os

# Hardcoded PDF file path
PDF_FILE = "COMPANY PROFILE - Reactive Space.pdf"

# SMTP Configuration (use your email credentials)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "haiqa.sheraz@gmail.com"
EMAIL_PASSWORD = "xxxx" //provide app password here

def send_email(recipient_email):
    subject = "Reactive Space Company Profile"
    body = "Please find attached the company profile for Reactive Space."
    
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    with open(PDF_FILE, "rb") as pdf:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(pdf.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(PDF_FILE)}",
        )
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Streamlit Application
st.title("Reactive Space")
st.write("We specialize in advanced AI and blockchain solutions. Let's innovate together!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Would you like to receive our company profile via email?"}
    ]

# Chat Interface
for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"] == "user"))

if user_input := st.text_input("Your message:"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    if user_input.lower() in ["yes", "sure", "ok"]:
        email = st.text_input("Please enter your email address:")
        if email:
            if send_email(email):
                st.success("The company profile has been sent to your email!")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "The profile has been sent to your email!"}
                )
            else:
                st.error("There was an error sending the email. Please try again later.")
    elif user_input.lower() in ["no", "not now"]:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Alright! Let us know if you change your mind."}
        )
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": "I didn't quite get that. Do you want the profile via email?"}
        )
