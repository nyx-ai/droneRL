from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import os
import uuid
from redis import Redis
from rq import Queue
from tasks import evaluate_submission

app = FastAPI()
redis_conn = Redis()
q = Queue(connection=redis_conn)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index():
    submissions = []
    try:
        submission_ids = redis_conn.lrange("submissions", 0, -1)
        for sid in submission_ids:
            sid = sid.decode("utf-8") if isinstance(sid, bytes) else sid
            data = redis_conn.hgetall(f"submission:{sid}")
            data = { k.decode("utf-8"): v.decode("utf-8") for k, v in data.items()}
            job = q.fetch_job(data.get("job_id"))
            if job:
                data["status"] = job.get_status()
                if job.is_finished:
                    data["result"] = str(job.result)
                redis_conn.hset(f"submission:{sid}", mapping={"status": data["status"], "result": data.get("result", "")})
            submissions.append(data)
    except Exception as e:
        print("Error loading submissions:", e)
    
    submissions_rows = ""
    for sub in submissions:
        submissions_rows += f"<tr><td>{sub.get('submission_id', '')}</td><td>{sub.get('job_id', '')}</td><td>{sub.get('participant_id', '')}</td><td>{sub.get('status', '')}</td><td>{sub.get('result', '')}</td></tr>"
    
    html_content = f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f7f7f7;
                }}
                h2 {{
                    color: #333;
                }}
                form {{
                    background-color: #fff;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: #fff;
                }}
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .refresh-button {{
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 20px;
                    cursor: pointer;
                    margin-bottom: 20px;
                }}
                .refresh-button:hover {{
                    background-color: #45a049;
                }}
            </style>
        </head>
        <body>
            <h2>Drone Evaluator Submission</h2>
            <form action="/submit" enctype="multipart/form-data" method="post">
                <label for="file">Upload Submission File:</label><br/>
                <input name="file" type="file" id="file" required/><br/><br/>
                <label for="aicrowd_participant_id">Participant ID (letters and numbers only):</label><br/>
                <input type="text" name="aicrowd_participant_id" id="aicrowd_participant_id" pattern="[A-Za-z0-9]+" title="Letters and numbers only" required/><br/><br/>
                <input type="submit" value="Submit" style="padding: 10px 20px;"/>
            </form>
            <button class="refresh-button" onclick="window.location.reload();">Refresh</button>
            <h2>Previous Submissions</h2>
            <table>
                <tr>
                    <th>Submission ID</th>
                    <th>Job ID</th>
                    <th>Participant ID</th>
                    <th>Status</th>
                    <th>Result</th>
                </tr>
                {submissions_rows}
            </table>
        </body>
    </html>
    """
    return html_content

@app.post("/submit")
async def submit_job(
    file: UploadFile = File(...),
    aicrowd_participant_id: int = Form(...)
):
    try:
        contents = await file.read()
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    submission_id = uuid.uuid4().hex
    client_payload = {
        "submission_file_path": file_path,
        "aicrowd_submission_id": submission_id,
        "aicrowd_participant_id": aicrowd_participant_id
    }
    job = q.enqueue(evaluate_submission, client_payload)

    redis_conn.hset(f"submission:{submission_id}", mapping={
        "job_id": job.get_id(), 
        "submission_id": submission_id,
        "participant_id": str(aicrowd_participant_id),
        "status": job.get_status(),
        "result": ""
    })
    redis_conn.rpush("submissions", submission_id)

    return {"submission_id": submission_id, "job_id": job.get_id(), "status": job.get_status()}

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    job = q.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = job.result if job.is_finished else None
    return {"job_id": job.id, "status": job.get_status(), "result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 