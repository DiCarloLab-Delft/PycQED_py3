// Pulse_fridge_sync
// trigger divider and distributor v1.00
// 2015/03/18
// by Raymond Vermeulen modified by Thijs Stavenga
// TU Delft TNW QT

#include <EEPROM.h>

//#define PSYNC PD2
//#define MARKER_OUT PD7
#define PULSE_CYCLE 24
#define BUF_LENGTH 64
#define MAX_PHASE 1000
#define MAX_DUTY 1000
#define ERR_BUF_OVFL -2
#define ERR_MISSED_PULSE -4

uint8_t const address = 29;
volatile uint8_t trigger = 0;
volatile uint8_t trigger_div_count = 0;
unsigned int trigger_error_detect_count = 0;
volatile uint8_t timer_done = 0;
volatile uint8_t phase_timer_done = 1;
volatile unsigned int phase_delay_ms;
unsigned int phase_count = 0;
volatile uint8_t mask_timer_done = 1;
volatile unsigned int mask_delay_ms;
unsigned int mask_count = 0;
signed int message_length = 0;
uint8_t missed_pulse = 0;
uint8_t i = 0;                     // iterator
// Function definitions
static uint8_t parse_command(char* ,uint8_t );



void setup()
{
  DDRD = DDRD | B11111000;  // Set pin D7 to D3 as output, leave the rest as is
  DDRD = DDRD & B11111011;  // Set pin D2 as input, leave the rest as is.
  PORTD = PORTD & B00000111;  // Set pin D7 to D3 to logic low, leave the rest as is
  PORTD = PORTD | B00000100;  // Set pin D2 as pullup, leave the rest as is
  PORTD |= (B10000000);

  cli();    // disable global interrupts
  TCCR1A = 0;    // make sure it's set to zero
  TCCR1B = B00001000;    // clear timer on compare match mode
  TIMSK1 = B00000010;    // generate interrupt when the value in OCR1A is reached
  OCR1A = 0x2BEB; // 45ms with the prescalar set to 64 0x15F8
  TCCR1B |= B00000011;    // start the timer and set prescaler to 64
  sei();    // enable global interrupts

  EICRA = EICRA | B00000010; // set INT0 in falling edge mode
  EIMSK = EIMSK | B00000001; // generate interrupt for INT0

  TCCR0A |= B00000011;  // Set the waveform generation mode to CTC (Clear on Timer Compare match mode)
  TCCR0B |= B00000000;  // Set the 64 prescale factor on Timer 0, leave the rest as is. This stops the clock for now
  TCCR0B |= B10000000;  // Set the Output compare mode to clear with compare match on Timer 0, leave the rest as is
  TIMSK0 |= B00000010;  // Set the Timer interrupt on match with OCR0A
  OCR0A   = 250;        // Set the output compare register to a value of 250. This yields a interrupt rate of 1kHz. so 1ms is min resolution

  TCCR2A |= B00000010;  // Set the waveform generation mode to CTC (Clear on Timer Compare match mode)
  TCCR2B |= B00000000;  // Set the 0 prescale factor on Timer 2, leave the rest as is. This stops the clock for now
  TCCR2A |= B10000000;  // Set the Output compare mode to clear with compare match on Timer 2, leave the rest as is
  TIMSK2 |= B00000010;  // Set the Timer interrupt on match with OCR2A
  OCR2A   = 250;        // Set the output compare register to a value of 250. This yields a interrupt rate of 1kHz. so 1ms is min resolution

  phase_delay_ms = EEPROMReadInt(address);
  mask_delay_ms = EEPROMReadInt(address+2);
  if(phase_delay_ms == 0 || phase_delay_ms > MAX_PHASE)
  {
    phase_delay_ms = 1;
  }
  if(mask_delay_ms == 0 || mask_delay_ms > MAX_DUTY)
  {
    mask_delay_ms = 40;
  }
  Serial.begin(115200);
}

void loop()
{
  if (trigger == 1)
  {
    if (trigger_div_count < PULSE_CYCLE)
    {
      trigger_div_count++;
    }
    else
    {trigger_div_count = 0;}
    if (trigger_error_detect_count == 0)
    {
    cli();    // disable global interrupts
    TCNT1 = 0;    // reset the counter
    sei();    // enable global interrupts
    }
    else
    {}
    trigger = 0;
    trigger_error_detect_count++;
  }
  else if (timer_done == 1)
  {
    if (trigger_error_detect_count != 2) // see if there were more or fewer than 1 triggers in this time period
    {
      missed_pulse = 1;
//      Serial.write(ERR_MISSED_PULSE);
      PORTB = PORTB | B00100000; // SET pin B5 high (turns the LED on), leave the rest
    }
    else
    {}  // leave all as is
    timer_done = 0;
    trigger_error_detect_count = 0;  // reset
  }

  while(Serial.available())
  {
    static char command_buffer[BUF_LENGTH];           // create a buffer for our message
    char data = Serial.read();  // copy the message into a buffer
    if(i> BUF_LENGTH)
    {
      i = 0; //Output Error
//      Serial.println(ERR_BUF_OVFL); //Bufer overflow error
    }
    command_buffer[i] = data;
    i++;
    if(data == '\r')
    {
      command_buffer[i] = '\0';
      uint8_t result = parse_command(command_buffer,i); // if our buffer is not empty, parse the command
      i = 0;
    }
  }
}

static uint8_t parse_command(char *cmdline, uint8_t len)
{
  char command[BUF_LENGTH];
  char* p_command;
  char* value;
  uint8_t query = 0;
  p_command = strsep(&cmdline," ");
  strcpy(command,p_command);
  if( p_command[strlen(p_command) -2]=='?' ) //there is a question mark
  {
    command[strlen(command) -2] = '\0';
    query = 1;
  }
  else // There is may be an argument
  {
    value = strsep(&cmdline,"");
    query = 0;
  }
  if (strcmp(command, "PHASE") == 0) {
    if(query)
    {
      Serial.println(phase_delay_ms);
    }
    else
    {
      unsigned int val = atoi(value);
      if (val < MAX_PHASE)
      {
        phase_delay_ms = val; //update true value and set into non volatile memory
        phase_timer_done = 1;
        EEPROMWriteInt(address, phase_delay_ms);
      }
      else
      {
        Serial.println(F("Phase value out of range: "));
        Serial.println(val);
      }

    }
  } else if (strcmp(command, "DUTY") == 0) {
    if(query)
    {
      Serial.println(mask_delay_ms);
    }
    else
    {
      unsigned int val = atoi(value);
      if (val < MAX_DUTY)
      {
        mask_delay_ms = atoi(value); //update true value and set into non volatile memory
        mask_timer_done = 1;
        EEPROMWriteInt(address+2, mask_delay_ms);
      }
      else
      {
        Serial.println(F("Duty cycle value out of range: "));
        Serial.println(val);
      }
    }
  } else if (strcmp(command, "*IDN") == 0) {
      if(query)
      Serial.println(F("Arduino Stroboscope\nAuthor: Thijs Stavenga\nPhone: 0639658542\nEmail: t.stavenga@tudelft.nl\n"));
  } else if (strcmp(command, "LOCK") == 0) {
      if(query)
      Serial.println(!(missed_pulse));
  }
  else  {
      Serial.println(F("Error: Unknown command: "));
      Serial.println(command);
  }
  return 0;
}

ISR(INT0_vect)
{
  trigger = 1;
  //at pulse 0 start the phase timer, however updating happens in the main interruptable loop. so check for 24
  if(trigger_div_count == PULSE_CYCLE)
  {
    if(phase_timer_done)
    {
      TCCR0B |= B00000011; // Start the phase timer
    }
    else // No pulse came in
    {
      TCCR0B &= !(B00000011);  // Turn off the clock
      TCNT0 = 0;               // reset the count
      phase_count = 0;
      TCCR0B |= B00000011; // Start the phase timer
    }
    phase_timer_done = 0;
  }
}

ISR (TIMER1_COMPA_vect)
{
  timer_done = 1;
}

ISR (TIMER0_COMPA_vect)
{
  phase_count++;
  if(phase_count >= phase_delay_ms)  // at 'phase' pulses start the Mask (Duty cycle) timer
  {
    if(mask_timer_done)
    {
      PORTD &= !(B10000000);      // Turn the mask on
      TCCR2B |= B00000100;     // Turn on the mask (Duty cycle) clock to a 64 prescaler
      TCCR0B &= !(B00000011);  // Turn off the phase clock until the new pulsetube clock pulse comes in
    }
    else // not finished, the duty cycle time is longer than the pulse tube period.
    {
      TCCR2B &= !(B00000100);  // Turn off the mask clock
      TCNT2 = 0;               // reset the count
      mask_count = 0;          // reset the mask count
      TCCR2B |= B00000100; // Start the mask timer
    }
    phase_count = 0;
    phase_timer_done = 1;
    mask_timer_done = 0;
  }
}

ISR (TIMER2_COMPA_vect)
{
  mask_count++;
  if(mask_count >= mask_delay_ms)  // at 'phase' pulses start the Mask (Duty cycle) timer
  {
    PORTD |= (B10000000);      // Turn on the mask (Duty cycle) clock. From now on no more measurements until next cycle
    TCCR2B &= !(B00000011);  // Turn off the clock until the new phase clock pulse arrives
    mask_count = 0;
    mask_timer_done = 1;
  }
}

void EEPROMWriteInt(int p_address, int p_value)
{
  byte lowByte = ((p_value >> 0) & 0xFF);
  byte highByte = ((p_value >> 8) & 0xFF);

  EEPROM.update(p_address, lowByte);
  EEPROM.update(p_address + 1, highByte);
}

//This function will read a 2 byte integer from the eeprom at the specified address and address + 1
unsigned int EEPROMReadInt(int p_address)
{
  byte lowByte = EEPROM.read(p_address);
  byte highByte = EEPROM.read(p_address + 1);

  return ((lowByte << 0) & 0xFF) + ((highByte << 8) & 0xFF00);
}

