    def create_gauge_chart(title, percentage, color):
        gauge = pygal.SolidGauge(
            half_pie=True,
            inner_radius=0.70,
            show_legend=False,
            style=custom_style
        )
        gauge.title = title  # Set title for the gauge
        gauge.add('', [{'value': percentage, 'max_value': 100, 'color': color}])
        return gauge.render(is_unicode=True)
        
    # Example data for three gauge charts
    titles = ['Arousal', 'Valence', 'Dominance']
    percentages = [percentageA, percentageV, percentageD]
    colors = [colourA, colourV, colourD]
    
    gauge_svgs = []
    for title, percentage, color in zip(titles, percentages, colors):
        gauge_svgs.append(create_gauge_chart(title, percentage, color))
    
    # Display gauge charts side by side in Streamlit
    #st.write("<div style='display:flex;'>")
    for gauge_svg in gauge_svgs:
        st.write(f"<div style='margin: auto;'>{gauge_svg}</div>", unsafe_allow_html=True)
    #st.write("</div>")
  
