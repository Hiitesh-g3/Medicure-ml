import google.generativeai as genai
genai.configure(api_key="AIzaSyCglOVm9tx5I2Er2zPfMDQkWHEPen3kdik")

models = genai.list_models()
for m in models:
    print(m.name)
