using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.IO.Ports;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ELM327_PID_DataCollector
{
    public class TcpClientOBD : SerialPort
    {
       
        private string port;

        private SerialPort serialPort;
        public bool connected = false;
        private bool forceStop = false;

        public delegate void EventPIDholder(string message);
        public event EventPIDholder PidMessageArrived;

        public delegate void EventPID();
        public event EventPID OBDdeviceReady;
        

        public TcpClientOBD(string _port)
        {
            port = _port;
        }


        public void StartOBDdev()
        {
            forceStop = false;
            bool Tryconnect = true;

            Task.Run(() =>
            {
                while (Tryconnect)
                {
                    try
                    {
                        serialPort = new SerialPort(port, 38400, Parity.None, 8, StopBits.One);
                        serialPort.Open();
                        Tryconnect = false;
                    }catch (SocketException e)
                    {
                        Console.WriteLine("Connection is not Successfull. Retrying...");
                    }catch(Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                }

                Console.WriteLine("Connected");



                connected = true;

                Task.Delay(1000).Wait();

                SetElm327Configs();

                Task.Delay(1000).Wait();

                OBDdeviceReady.Invoke();

                string data = "";
                int k = 0;

                
                   
                
            });
        }


        public void StartOBDdevFreeMode()
        {
            forceStop = false;
            bool Tryconnect = true;

            while (Tryconnect)
            {
                try
                {
                    serialPort = new SerialPort(port, 38400, Parity.None, 8, StopBits.One);
                    serialPort.Open();
                    Tryconnect = false;
                }
                catch (SocketException e)
                {
                    Console.WriteLine("Connection is not Successfull. Retrying...");
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }

            Console.WriteLine("Connected");



            connected = true;

            Task.Delay(1000).Wait();

            SetElm327Configs();

            Task.Delay(1000).Wait();

            Task.Run(() =>
            {
                string data = "";
                int k = 0;

                while (!forceStop)
                {
                    // Buffer to store the response bytes.
                    try
                    {
                        string response = serialPort.ReadExisting();

                        byte[] mesajj = Encoding.Default.GetBytes(response);


                        k++;
                        data += Encoding.Default.GetString(mesajj, 0, mesajj.Length).Replace("\r", " ");

                        if (data.EndsWith('>') || data.Length > 128 || k > 10)
                        {

                            k = 0;
                            if (data.Length > 128)
                            {
                                send("\r");
                            }

                            PidMessageArrived.Invoke(data);

                            data = "";
                        }
                    }
                    catch (Exception e)
                    {

                        //Console.WriteLine(e.Message);
                        //Console.WriteLine("Connection Lost! Trying Connect Again");
                        //forceStop= true;
                        //StartOBDdev();
                    }
                }
            });

            OBDdeviceReady.Invoke();

            


        }

        void CheckObdConnection()
        {
            


        }

        void SetElm327Configs()
        {

            send("\r");

            send("AT SP 0" + "\r");

            send("AT D" + "\r");

            send("AT DPN" + "\r");

            send("AT I" + "\r");

            send("0100" + "\r");

            send("AT H0" + "\r");

            send("AT AT2" + "\r");

            send("AT SH 7FF" + "\r");

            Console.WriteLine("Battery Voltage:");
            send("AT RV" + "\r");

            send("0902" + "\r");
        }

        public void SendSpeedRequest()
        {
            send("010D" + "\r");
        }

        public void SendRpmRequest()
        {
            send("010C" + "\r");
        }

        public void SendFuelLevelRequest()
        {
            send("012F" + "\r");
        }

        public void SendRequest(string id,string mode)
        {
            send(mode+id + "\r");
        }

        public void send(byte[] msg)
        {

            Console.WriteLine("SENT");
            try
            {
                serialPort.Write(msg, 0, msg.Length);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }

        public void send(string msg)
        {
            
            var msgByte = Encoding.ASCII.GetBytes(msg);
            try
            {
                serialPort.Write(msgByte,0,msg.Length);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            
        }

        public void Stop()
        {
            forceStop= true;
            Dispose();
            serialPort.Close();
        }
    }
}
