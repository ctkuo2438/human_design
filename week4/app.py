# sudo su - ec2-user

# sudo yum install python3-pip -y
# pip3 install streamlit boto3

# start: streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# cat > app.py << 'EOF'
import streamlit as st
import boto3
import json
import base64

# AWS Lambda client
lambda_client = boto3.client("lambda", region_name="us-east-1")

# API Gateway base URL
API_BASE = "https://vemefih3xa.execute-api.us-east-1.amazonaws.com/prod"

st.title("Human Design Reading System")

tab1, tab2 = st.tabs(["Upload Chart Image", "Ask a Question"])

# Tab 1: Upload image → Vision → RAG Reading
with tab1:
    uploaded_file = st.file_uploader("Upload your BodyGraph chart", type=["png", "jpg", "jpeg"])
    query = st.text_input("Ask a question about your chart (optional)", key="tab1_query")
    
    if st.button("Get Reading", key="tab1_btn"):
        if not uploaded_file:
            st.error("Please upload a BodyGraph image.")
        else:
            # Step 1: Call Vision Lambda via API Gateway
            with st.spinner("Analyzing your chart..."):
                import requests
                img_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
                media_type = f"image/{uploaded_file.type.split('/')[-1]}"
                
                vision_resp = requests.post(
                    f"{API_BASE}/vision",
                    json={"image": img_b64, "media_type": media_type},
                    timeout=30
                )
                vision_result = vision_resp.json()
            
            if not vision_result.get("success"):
                st.error(f"Vision extraction failed: {vision_result.get('error')}")
            else:
                chart_data = json.loads(vision_result["body"])["chart_data"] if isinstance(vision_result.get("body"), str) else vision_result.get("chart_data")
                st.subheader("Extracted Chart Data")
                st.json(chart_data)
                
                # Step 2: Call RAG Lambda via boto3 (no 29s timeout limit)
                with st.spinner("Generating your personalized reading... (this may take 1-3 minutes on first run)"):
                    rag_payload = {"chart_data": chart_data}
                    if query:
                        rag_payload["query"] = query
                    
                    rag_resp = lambda_client.invoke(
                        FunctionName="human-design-rag",
                        InvocationType="RequestResponse",
                        Payload=json.dumps(rag_payload)
                    )
                    rag_result = json.loads(rag_resp["Payload"].read().decode("utf-8"))
                    rag_body = json.loads(rag_result["body"]) if isinstance(rag_result.get("body"), str) else rag_result
                
                if rag_body.get("success"):
                    st.subheader("Your Personalized Reading")
                    st.markdown(rag_body["reading"])
                    
                    with st.expander("Sources"):
                        for s in rag_body.get("sources", []):
                            st.write(f"- {s['source']} (score: {s['score']:.3f})")
                else:
                    st.error(f"Reading generation failed: {rag_body.get('error')}")

# Tab 2: Query only → RAG Reading
with tab2:
    user_query = st.text_input("Ask anything about Human Design", key="tab2_query")
    
    if st.button("Ask", key="tab2_btn"):
        if not user_query:
            st.error("Please enter a question.")
        else:
            with st.spinner("Searching knowledge base and generating answer... (this may take 1-3 minutes on first run)"):
                rag_resp = lambda_client.invoke(
                    FunctionName="human-design-rag",
                    InvocationType="RequestResponse",
                    Payload=json.dumps({"query": user_query})
                )
                rag_result = json.loads(rag_resp["Payload"].read().decode("utf-8"))
                rag_body = json.loads(rag_result["body"]) if isinstance(rag_result.get("body"), str) else rag_result
            
            if rag_body.get("success"):
                st.subheader("Answer")
                st.markdown(rag_body["reading"])
                
                with st.expander("Sources"):
                    for s in rag_body.get("sources", []):
                        st.write(f"- {s['source']} (score: {s['score']:.3f})")
            else:
                st.error(f"Failed: {rag_body.get('error')}")
# EOF