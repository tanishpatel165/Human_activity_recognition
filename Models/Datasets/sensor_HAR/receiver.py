from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/data")
async def data(request: Request):
    body = await request.json()
    for item in body.get("payload", []):
        print(item.get("name"), item.get("values"))
    return {"ok": True}