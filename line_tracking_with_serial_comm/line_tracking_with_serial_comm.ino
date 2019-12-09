//Line Tracking IO
#define LT_R !digitalRead(10) // rightmost IR LED is blocked
#define LT_M !digitalRead(4)  // middle IR LED is blocked
#define LT_L !digitalRead(2)  // leftmost IR LED is blocked

#define ENB 5
#define IN1 7
#define IN2 8
#define IN3 9
#define IN4 11
#define ENA 6

int carSpeed = 200;

void forward(){ // sets the outputs on the car for forward motion
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  Serial.println("go forward!");
}

void back(){  // sets the outputs on the car for backwards motion
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  Serial.println("go back!");
}

void left(){  // sets the outputs on the car for lefward motion
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  Serial.println("go left!");
}

void right(){ // sets the outputs on the car for rightward motion
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW); 
  Serial.println("go right!");
} 

void stop(){ // sets the outputs on the car to have it stop
   digitalWrite(ENA, LOW);
   digitalWrite(ENB, LOW);
   Serial.println("Stop!");
} 

void setup(){
  Serial.begin(9600); // set up the baudrate
}

void loop() {
  if(Serial.available() > 0) { // if commnnication line is open
    if(Serial.read() == 's') { // if we receive a stop command
      stop(); // stop the car
      delay(2000); // wait 2 seconds before continuing on
    }
  }

  if(LT_R) { // if rightmost IR LED is blocked, tape is on right of middle of car
    right(); // turn right until the car is centered
    while(LT_R);
  }
  else if(LT_L) { // if leftmost IR LED is blocked, tape is on left of middle of car
    left();  // turn left until the car is centered
    while(LT_L);
  } else if(LT_M){ // if middle IR LED is blocked, car is oriented properly
    forward(); // continue forward
  }
}
