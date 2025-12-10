# Beat-the-Bet
### Data Science group project

##### Instructions for importing game data from basketball-reference.com 
1. Go to basketball-reference.com
2. Find "Scores" in the bar at the top
3. Change date to the day you want to import the game stats form
4. Find the game that you're looking for
5. Select "Box Score"
6. Scroll down to see "Team A Basic and Advanced Stats"
7. Click "Share & Export"
8. Click "Get table as CSV (for Excel)
9. Copy everything including and below the row that says "Starters, MP, FG, ..."
10. Find folder in this repo for correct team
11. Create a new file inside the folder titled "TeamNameMonth.Day.Year.csv"
12. Paste info
13. Commit
14. Go back to basketball-reference and go to the table for the advanced stats for the same team (MP, TS%, eFG%)
15. Repeat steps 7-10
16. Create a new file inside the folder titled "TeamNameMonth.Day.Yearadv.csv"
17. Paste advnaced stats
18. Commit
19. Repeat steps 6-18 for Team B
20. Game is fully imported

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

##### HOW TO RUN
1. run "pip install -r requirements.txt"
2. run "python3 -m src.dataset"
3. run "python3 -m src.XGBoostModel"
4. run "python3 -m src.predict "team1" "team2"
5. If it tells you to install something in between just "pip install <package>"
