# utils.py
import streamlit as st
import os
import base64

def display_custom_sidebar():
    # Inject custom CSS for sidebar styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        /* Style the Streamlit sidebar */
        div[data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #1e3c72, #2a5298);
            color: white;
            padding: 20px;
            box-sizing: border-box;
            width: 220px !important;
        }
        /* Hide the stSidebarNav section */
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }
        /* Ensure stSidebarUserContent takes full space */
        div[data-testid="stSidebarUserContent"] {
            padding: 0 !important;
        }
        .app-title {
            font-size: 24px;
            font-weight: 600;
            text-align: center;
            padding: 20px 0;
            background: -webkit-linear-gradient(#f8f9fa, #c9d6df);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-item {
            display: flex;
            align-items: center;
            padding: 12px 15px;
            margin: 8px 0;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            text-decoration: none;
            color: white;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .nav-item:hover {
            background-color: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
        .nav-item.active {
            background-color: rgba(255, 255, 255, 0.25);
            border-left: 4px solid #4CAF50;
        }
        .nav-icon {
            margin-right: 12px;
            font-size: 18px;
        }
        .sidebar-separator {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.3), rgba(255,255,255,0));
            margin: 20px 0;
        }
        .sidebar-footer {
            text-align: center;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.6);
            background-color: rgba(0, 0, 0, 0.3);
            padding: 15px;
            margin-top: 20px;
        }
        .main-content {
            margin-left: 0;
            padding: 20px;
        }
        header { visibility: hidden; }
        section.main > div:first-child { padding-top: 0 !important; }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for the selected page if not already set
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "dashboard"

    # Determine the active page from session state
    page_path = st.session_state.selected_page

    # Display the current selected page in the sidebar



    def get_image_base64(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return encoded

    logo_path = os.path.join("images", "Fraud_logo.png")
    logo_base64 = get_image_base64(logo_path)

    with st.sidebar:
        st.markdown(f"""
        <div class="app-title">
            <img src="data:image/png;base64,{logo_base64}" 
                style="width: 100%; max-width: 150px; display: block; margin: 0 auto 10px;">
        </div>
        """, unsafe_allow_html=True)

        # Navigation items
        nav_items = [
            ("🏠 Home Page ", "dashboard"),
            ("📤 Batch Detection", "batch_fraud_detection"),
            ("🔍 Transaction Checker", "transaction_checker"),
            ("📈 Fraud Insights", "fraud_insights")
        ]
        for label, page in nav_items:
            is_active = "active" if page_path == page else ""
            # Render navigation item as a clickable element
            if st.button(
                label,
                key=f"sidebar_btn_{page}_{label}",
                help=f"Switch to {label.split(maxsplit=1)[1]}",
                use_container_width=True
            ):
                st.session_state.selected_page = page
                st.write(f"Debug: Switching to {page}")  # Debug message
                st.rerun()  # Rerun the app to update the content

            # Apply custom styling to the button to match the nav-item look
            st.markdown(
            f"""
            <style>
            button[kind="primary"][key="sidebar_btn_{page}"] {{
                background: none !important;
                border: 0 !important;  /* Remove all borders */
                padding: 12px 15px !important;
                margin: 8px 0 !important;
                background-color: {'rgba(255, 255, 255, 0.25)' if is_active else 'rgba(255, 255, 255, 0.1)'} !important;
                border-radius: 10px !important;
                color: white !important;
                text-align: left !important;
                display: flex !important;
                align-items: center !important;
                transition: all 0.3s ease !important;
                position: relative !important;  /* For pseudo-element positioning */
            }}
            button[kind="primary"][key="sidebar_btn_{page}"]:hover {{
                background-color: rgba(255, 255, 255, 0.2) !important;
                transform: translateX(5px) !important;
            }}
            button[kind="primary"][key="sidebar_btn_{page}"] > div {{
                display: flex !important;
                align-items: center !important;
            }}
            button[kind="primary"][key="sidebar_btn_{page}"] p {{
                margin: 0 !important;
                font-family: 'Poppins', sans-serif !important;
            }}
            button[kind="primary"][key="sidebar_btn_{page}"]::before {{
                content: "{label.split()[0]}";
                margin-right: 12px !important;
                font-size: 18px !important;
            }}
            /* Add green indicator for active state using a pseudo-element */
            button[kind="primary"][key="sidebar_btn_{page}"][data-active="true"]::after {{
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                height: 100%;
                width: 4px;
                background-color: #4CAF50;
                border-radius: 10px 0 0 10px;
            }}
            </style>
            """,
            unsafe_allow_html=True
            )

    # Custom sidebar footer
    custom_sidebar_footer = """
        <style>
            /* Target Streamlit sidebar */
            [data-testid="stSidebar"] {
                position: relative;
            }

            .custom-footer {
                position: absolute;
                bottom: -160px;  /* Adjust this value to move the footer up or down */
                width: 100%;
                text-align: center;
                font-size: 13px;
                padding: 10px 0;
                color: #aaa;
            }
        </style>
        <div class="custom-footer">© 2025 FYP tanshiying</div>
    """
    # Inject footer only into the sidebar
    with st.sidebar:
        st.markdown(custom_sidebar_footer, unsafe_allow_html=True)

    # Start main content area
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Return a function to close the main content div
    def close_content():
        st.markdown('</div>', unsafe_allow_html=True)

    return close_content