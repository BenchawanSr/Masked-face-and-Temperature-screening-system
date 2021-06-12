#include <Servo.h>
#include <Wire.h>
Servo myservo;
Servo myservo2;
char key;
int dirmotor = 1;
int dl = 5000;
void setup()
{ 
  myservo.attach(D8);
  //myservo2.attach(D7);
  Serial.begin(9600);
  myservo.write(0);
  //myservo2.write(0);
  delay(dl);
}
void loop()
{
  if (Serial.available())
  {
    key = Serial.read();
    if (key == 'O')
    {
      myservo.write(90);
      //myservo2.write(90);
      delay(dl);
      myservo.write(0);
      //myservo2.write(0);
      delay(5000);
      //dirmotor = -1;
    }
  }
}
