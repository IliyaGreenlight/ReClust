import json
from googleapiclient.discovery import build
from dateutil import parser
from dateutil.relativedelta import relativedelta


def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url.rstrip("/").split("/")[-1]


def format_relativedelta(rd: relativedelta) -> str:
    parts = []
    if rd.years:   parts.append(f"{rd.years}y")
    if rd.months:  parts.append(f"{rd.months}mo")
    if rd.days:    parts.append(f"{rd.days}d")
    if rd.hours:   parts.append(f"{rd.hours}h")
    if rd.minutes: parts.append(f"{rd.minutes}m")
    if rd.seconds: parts.append(f"{rd.seconds}s")
    return " ".join(parts) or "0s"


def load_comments_from_youtube(api_key: str, video_url: str):
    if not api_key or not video_url:
        raise SystemExit("API key and video URL must be provided.")

    video_id = extract_video_id(video_url)
    youtube = build("youtube", "v3", developerKey=api_key)

    video_resp = youtube.videos().list(part="snippet", id=video_id).execute()
    if not video_resp.get("items"):
        raise SystemExit("Video not found or invalid video id.")
    video_publish_time = parser.isoparse(video_resp["items"][0]["snippet"]["publishedAt"])

    comments = []
    next_page_token = None

    while True:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in resp.get("items", []):
            c = item["snippet"]["topLevelComment"]["snippet"]
            comment_id = item["snippet"]["topLevelComment"]["id"]
            reply_count = item["snippet"].get("totalReplyCount", 0)

            # Skip empty or whitespace-only comments early
            body = c.get("textDisplay", "")
            if not body or not body.strip():
                continue

            comment_time = parser.isoparse(c["publishedAt"])
            delta = comment_time - video_publish_time
            rd = relativedelta(comment_time, video_publish_time)

            # --- Top-level comment ---
            top_comment = {
                "type": "comment",
                "id": comment_id,
                "author_name": c.get("authorDisplayName", "Unknown"),
                "body": c.get("textDisplay", ""),
                "like_count": c.get("likeCount", 0),
                "reply_count": reply_count,
                "comment_published_at": c.get("publishedAt"),
                "minutes_since_video_published": int(delta.total_seconds() / 60),
                "time_since_video_published": format_relativedelta(rd),
                "replies": []
            }

            # --- Fetch replies if they exist ---
            if reply_count > 0:
                replies_next_page = None
                while True:
                    reply_resp = youtube.comments().list(
                        part="snippet",
                        parentId=comment_id,
                        maxResults=100,
                        pageToken=replies_next_page,
                        textFormat="plainText"
                    ).execute()

                    for reply in reply_resp.get("items", []):
                        rc = reply["snippet"]
                        reply_time = parser.isoparse(rc["publishedAt"])
                        delta_r = reply_time - video_publish_time
                        rd_r = relativedelta(reply_time, video_publish_time)

                        top_comment["replies"].append({
                            "type": "reply",
                            "id": reply["id"],
                            "author_name": rc.get("authorDisplayName", "Unknown"),
                            "body": rc.get("textDisplay", ""),
                            "like_count": rc.get("likeCount", 0),
                            "comment_published_at": rc.get("publishedAt"),
                            "minutes_since_video_published": int(delta_r.total_seconds() / 60),
                            "time_since_video_published": format_relativedelta(rd_r)
                        })

                    replies_next_page = reply_resp.get("nextPageToken")
                    if not replies_next_page:
                        break

            comments.append(top_comment)

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    with open("data/comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)