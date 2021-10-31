import smtplib
import ssl


def send_mail(
    self,
    broadcaster_handler,
    receiver_handler,
    password_handler,
):
    """
    Method of sending e-mails to list of receivers by the broadcaster
    (e-mail) using a special browser key (each argument is in a
    different files for privacy and enable easy extension).

    Parameters:
    - broadcaster_handler (string) - path to the file where the
    sender's e-mail is saved
    - receiver_handler (string) - path to the file where the receiver's
    e-mails are saved, separeted with ';'
    - password_handler (string) - path to the file, where password is saved
    """
    port = 465
    smtp_serv = "smtp.gmail.com"
    try:
        broadcaster = open(broadcaster_handler).read()
        receiver = open(receiver_handler).read().split(";")
        password = open(password_handler).read()
        message = self.raport_to_mail()
        del receiver[-1]

        ssl_pol = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_serv, port, context=ssl_pol) as serwer:
            serwer.login(broadcaster, password)
            serwer.sendmail(broadcaster, receiver, message)
    except Exception as e:
        print("Mail sending error:", e)
    else:
        print("Mail was sent!")
