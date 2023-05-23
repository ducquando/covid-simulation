import streamlit as st
from SIRmodel import SIR

@cache_on_button_press('Generate Covid network')
def generate_video_button(network, population, time, rate_si, rate_ir):
    covid = SIR(network = network, population = population, time = time, rate_si = rate_si, rate_ir = rate_ir)
    video = covid.make_video("simulation.mp4")
    
    return video


def main():
    with st.sidebar:
        st.header("Simulation options")
        
        # Choose network type
        network_options = ("Scale-free", "Small-world", "Random")
        network_selected = st.selectbox(
            label = "Select network type",
            options = network_options,
        )
        
        # Select population size
        population_selected = st.slider('Population size', 200, 2000, 1000, 200)
        
        # Select time steps
        time_selected = st.slider('Select # of days', 10, 30, 30, 10)

        # Select time steps
        time_selected = st.slider('Select # of days', 10, 30, 30, 10)
        
        # Select rate si
        rate_si_selected = st.slider('Select Susceptible to Infectious rate', 0.3, 0, 1, 0.001)
        
        # Select rate ir
        rate_ri_selected = st.slider('Select Infectious to Recovered rate', 0.3, 0, 1, 0.178)
        
        # Run
        video_bytes = generate_video_button(network_selected, population_selected, time_selected, rate_si_selected, rate_ri_selected):

    # Display
    st.title("Covid Network Simulation")
    st.video(video_bytes, format = "video/mp4", start_time = 0)
        

if __name__ == "__main__":
    st.set_page_config(page_title = "Covid Simulation", page_icon = ":chart_with_upwards_trend:")
    main()
