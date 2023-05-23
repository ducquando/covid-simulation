import streamlit as st
from SIRmodel import SIR

def main():
    # Init model
    pressed = False
    
    with st.sidebar:
        st.header("Simulation options")
        
        # Choose network type
        network_options = ("Scale-free", "Small-world", "Random")
        network_selected = st.selectbox(label = "Network type", options = network_options)
        
        # Select population size
        population_selected = st.slider('Population size', 200, 2000, 400, 200)
        
        # Select time steps
        time_selected = st.slider('Number of days', 10, 30, 30, 5)
        
        # Select rate si
        rate_si_selected = st.slider('Susceptible to Infectious rate', 0.0, 1.0, 0.30, 0.01)
        
        # Select rate ir
        rate_ri_selected = st.slider('Infectious to Recovered rate',  0.0, 1.0, 0.18, 0.01)
        
        # Change model once pressed
        if st.button('Generate'):
            pressed = True 
            
    
    # Title
    st.title("Covid Network Simulation")

    # Display graphs
    if pressed:
        covid = SIR(network_type = network_selected, population = population_selected,
                    time = time_selected, rate_si = rate_si_selected, rate_ir = rate_ri_selected, is_online = True)

        # Distribution
        st.subheader("Personal network degree distribution")
        img_bytes = covid.make_histogram_online()
        st.image(img_bytes)
            
        # Video
        st.subheader("Covid simulation")
        video_bytes = covid.make_video_online("simulation.mp4")
        st.video(video_bytes, format = "video/mp4", start_time = 0)
        

if __name__ == "__main__":
    st.set_page_config(page_title = "Covid Simulation", page_icon = ":chart_with_upwards_trend:")
    main()
