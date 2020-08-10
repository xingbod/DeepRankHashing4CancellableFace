% function mailTome(subject,content)
% %when done, send mail to me
% 
% setpref('Internet','SMTP_Server','smtp.gmail.com');
% setpref('Internet','E_mail','xingbod@gmail.com');
% setpref('Internet','SMTP_Username','xingbod@gmail.com');
% setpref('Internet','SMTP_Password','XXXX');
% props = java.lang.System.getProperties;
% props.setProperty('mail.smtp.auth','true');
% props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
% props.setProperty('mail.smtp.socketFactory.port','465');
% sendmail('xingbod@gmail.com',subject,content);
% 
% 
% 
% end