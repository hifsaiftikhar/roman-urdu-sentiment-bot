import scrapetube
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import json, os, re, time, shutil
from datetime import datetime

ROMAN_URDU_WORDS = [
    'hai', 'hain', 'nahi', 'nahin', 'kya', 'aur', 'bhi', 'toh',
    'kar', 'tha', 'thi', 'wala', 'wali', 'yaar', 'bhai', 'achi',
    'bekar', 'bilkul', 'bohat', 'bahut', 'mujhe', 'kuch', 'sab',
    'mera', 'achha', 'acha', 'bura', 'theek', 'thik', 'zabardast',
    'bakwas', 'yeh', 'ye', 'wo', 'woh', 'aap', 'main', 'hum',
    'phir', 'lekin', 'magar', 'kyun', 'matlab', 'lagta', 'lagti',
    'zyada', 'thoda', 'bilkul', 'abhi', 'sirf', 'bas', 'agar',
    'pakka', 'waise', 'aaya', 'gayi', 'gaya', 'raha', 'hoga',
    'karo', 'karna', 'dekho', 'liya', 'diya', 'video', 'order',
    'delivery', 'return', 'account', 'app', 'course', 'channel',
    'problem', 'mobile', 'online', 'screen', 'button', 'update'
]

PROFANITY = [
    'madarchod', 'kutte ke', 'haramzade', 'chutiya',
    'behenchod', 'benchod', 'gaand'
]

SPAM = [r'http\S+|www\S+', r'[!]{4,}', r'[?]{4,}', r'\d{10,}']


def is_roman_urdu(text):
    words = text.lower().split()
    if not words:
        return False
    matches = sum(1 for w in ROMAN_URDU_WORDS if w in words)
    return matches >= 2


def clean(text):
    for p in SPAM:
        text = re.sub(p, '', text)
    return ' '.join(text.split()).strip()


def is_good(text):
    if len(text) < 15 or len(text) > 500:
        return False
    if sum(c.isalpha() for c in text) < 8:
        return False
    tl = text.lower()
    if any(p in tl for p in PROFANITY):
        return False
    return True


def fetch_comments(keyword, max_comments=300, use_cache=True):
    cache_file = f"cache/{keyword.replace(' ','_').lower()}.json"
    os.makedirs("cache", exist_ok=True)

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[CACHE] Loaded {len(data['comments'])} comments")
        return data['comments']

    print(f"\nSearching: '{keyword} Pakistan'")

    videos = list(scrapetube.get_search(keyword + " Pakistan", limit=20))
    print(f"Found {len(videos)} videos\n")

    downloader = YoutubeCommentDownloader()
    all_comments = []
    seen = set()

    for i, video in enumerate(videos):
        if len(all_comments) >= max_comments:
            break

        vid_id = video['videoId']
        title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
        url = f"https://www.youtube.com/watch?v={vid_id}"
        print(f"  [{i+1}/{len(videos)}] {title[:55]}...")

        try:
            comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
            count = 0
            for c in comments:
                txt = clean(c['text'].strip())
                if txt in seen:
                    continue
                if not is_good(txt):
                    continue
                if not is_roman_urdu(txt):
                    continue
                all_comments.append(txt)
                seen.add(txt)
                count += 1
                if count >= 50:
                    break
            print(f"         → {count} collected")
            time.sleep(1)
        except Exception as e:
            print(f"         → Failed: {e}")

    print(f"\nTOTAL: {len(all_comments)} comments")

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({'keyword': keyword, 'timestamp': datetime.now().isoformat(),
                   'comments': all_comments}, f, ensure_ascii=False, indent=2)

    return all_comments


if __name__ == "__main__":
    if os.path.exists("cache"):
        shutil.rmtree("cache")
        print("[CACHE] Cleared\n")

    comments = fetch_comments("Daraz", max_comments=300)

    print("\nSAMPLE COMMENTS:")
    print("-" * 50)
    for i, c in enumerate(comments[:10]):
        print(f"{i+1}. {c}\n")

    print(f"Total: {len(comments)}")