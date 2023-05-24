from SIRmodel import SIR

def main():
   # Set params
   networks = ["random", "scale-free", "small-world"]
   POPULATION = 300000
   TIME = 30 # days
   RATE_SI = 0.00000360918393
   RATE_RI = 0.178

   # Random network without quarantine
   rd = SIR(networks[0], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI)
   rd.make_histogram()
   rd.make_video("simulation.mp4")

   # Small world network without quarantine
   sw = SIR(networks[2], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI)
   sw.make_histogram()
   sw.make_video("simulation.mp4")

   # Scale-free network without quarantine
   sf1 = SIR(networks[1], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI)
   sf1.make_histogram()
   sf1.make_video("simulation.mp4")

   # Scale-free network with quarantine
   sf2 = SIR(networks[1], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI, folder = "quarantine", is_quarantine = True)
   sf2.make_video("quarantine.mp4")

if __name__ == "__main__":
   main()
