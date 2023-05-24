import streamlit as st
from SIRmodel import SIROnline

def main():
    # Init model
    pressed = False
    
    with st.sidebar:
        st.header("Simulation options")
        
        # Choose network type
        network_options = ("scale-free", "small-world", "random")
        network_selected = st.selectbox(label = "Network type", options = network_options)
        
        # Select population size
        st.caption("We suggest choosing a small population size (e.g. 400-1000) for fast computation")
        population_selected = st.slider('Population size', 200, 300000, 400, 200)
        
        # Select time steps
        time_selected = st.slider('Number of days', 10, 30, 30, 5)
        
        # Select rate si
        
        rate_si_selected = st.number_input('Susceptible to Infectious rate', 0.0, 1.0, 0.00000360918393)
        
        # Select rate ir
        rate_ri_selected = st.number_input('Infectious to Recovered rate',  0.0, 1.0, 0.18)

        # Choose quarantine
        quarantine_selected = st.checkbox('Quarantine?')
        
        # Change model once pressed
        if st.button('Generate'):
            pressed = True 
            
    
    # Title
    st.title("Covid Network Simulation")

    # Display graphs
    if pressed:
        covid = SIROnline(network_type = network_selected, population = population_selected,
                          time = time_selected, rate_si = rate_si_selected, rate_ir = rate_ri_selected, is_online = True, is_quarantine = quarantine_selected)

        # Distribution
        st.subheader("Personal network degree distribution")
        img_bytes = covid.make_histogram()
        st.image(img_bytes)
            
        # Video
        st.subheader("Covid simulation")
        video_bytes = covid.make_video("simulation.mp4")
        st.video(video_bytes, format = "video/mp4", start_time = 0)
        

if __name__ == "__main__":
    st.set_page_config(page_title = "Covid Simulation", page_icon = ":chart_with_upwards_trend:")
    main()
