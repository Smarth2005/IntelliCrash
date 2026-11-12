# IntelliCrash login / register — setup

## 1. Install the one dependency
    npm install better-sqlite3
(If Next.js complains about it, add to next.config.js:  serverExternalPackages: ["better-sqlite3"]
 — on Next 14 use experimental.serverComponentsExternalPackages instead.)

## 2. Copy the files
- AuthScreen.jsx          -> same folder as your Dashboard.jsx
- Dashboard.jsx           -> replace your existing one (same folder)
- api/auth/  (whole folder) -> <your app folder>/api/auth/
      e.g. src/app/api/auth/register/route.js
           src/app/api/auth/_db.js ...

## 3. Run
    npm run dev
Open the site -> you will see the login screen. Click "Register", create an account, and you are in.
Users are stored in  <project>/data/intellicrash.db  (created automatically).
Add  data/intellicrash.db*  to .gitignore.

## Handy during the demo
View users:   sqlite3 data/intellicrash.db "select id,name,email,role,created_at from users;"
Reset all:    stop the server, delete data/intellicrash.db*

## Owner & vehicle details (new)
- New folder  api/profile/  (copy next to api/auth/, e.g. src/app/api/profile/route.js).
- _db.js now also creates an `owner_profiles` table automatically (no manual migration).
- The Vehicle & Health tab, the owner drawer and the Live Cockpit plate all read from the signed-in
  Fleet Owner's profile. Owners click the bottom-left profile -> "Edit owner & vehicle details".
- Responders / engineers see the first registered owner's vehicle + medical card (read-only).
- Existing users keep working; their profile row is created on first load.
