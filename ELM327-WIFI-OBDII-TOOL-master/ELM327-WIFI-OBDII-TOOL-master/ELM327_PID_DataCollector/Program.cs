using ELM327_PID_DataCollector.Helpers;
using System.Reflection;
using System.IO;
using System.Reflection;
using ELM327_PID_DataCollector.Items;

namespace ELM327_PID_DataCollector
{
    internal class Program
    {

        static void Main(string[] args)
        {
            Console.WriteLine("\r\n  ___  _     __  __  ____ ___  ____  __      __ _  ___  _      _  _  ___  _____ \r\n | __|| |   |  \\/  ||__ /|_  )|__  | \\ \\    / /(_)| __|(_)    | \\| || __||_   _|\r\n | _| | |__ | |\\/| | |_ \\ / /   / /   \\ \\/\\/ / | || _| | |  _ | .` || _|   | |  \r\n |___||____||_|  |_||___//___| /_/     \\_/\\_/  |_||_|  |_| (_)|_|\\_||___|  |_|  \r\n                                                                                \r\n");
            Console.WriteLine();
            Console.Write("Enter OBD Device COM PORT: ");
            var port = Console.ReadLine();
  

            Console.WriteLine("***************************");


            Elm327wifi elmObd = new Elm327wifi(port);

            while (true)
            {
                elmObd.Start();


                elmObd.Stop();
            }
        }

    }
}