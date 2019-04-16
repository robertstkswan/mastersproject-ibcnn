#!/bin/bash
sudo apt install ssmtp
sudo echo "root=message.prompt.ag939@gmail.com" >> /etc/ssmtp/ssmtp.conf
sudo echo "mailhub=smtp.gmail.com:465" >> /etc/ssmtp/ssmtp.conf
sudo echo "rewriteDomain=gmail.com" >> /etc/ssmtp/ssmtp.conf
sudo echo "AuthUser=message.prompt.ag939" >> /etc/ssmtp/ssmtp.conf
sudo echo "AuthPass=$1" >> /etc/ssmtp/ssmtp.conf
sudo echo "FromLineOverride=YES" >> /etc/ssmtp/ssmtp.conf
sudo echo "UseTLS=YES" >> /etc/ssmtp/ssmtp.conf
