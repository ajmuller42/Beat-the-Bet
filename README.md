# Beat-the-Bet
### Data Science group project

##### NOTE
Some team files do not use the full name bc I don't want to type out the full name
List of abbreviations:
Trailblazers -> Blazers
Mavericks -> Mavs
76ers -> Sixers (I just felt like not using numbers in the title of files next to date numbers)

##### NBA stats API
https://github.com/swar/nba_api

##### API Table of Contents
https://github.com/swar/nba_api/blob/master/docs/table_of_contents.md

##### To install API:
pip install nba_api

##### HOW TO RUN (ON VS CODE)
1. cd server
2. pip install -r requirements.txt
3. cd ..
4. uvicorn server.api:app --reload
5. If this throws an error, try this line instead: python -m uvicorn server.api:app --reload
6. Open the extensions tab on VS code and search up "Live Server"
7. Install "Live Server" by Ritwick Dey
8. Expand the frontend directory on the left-hand bar and right-click on index.html
9. Click "Open with Live Server"
