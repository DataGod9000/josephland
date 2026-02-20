# Deploy Joseph Land to the web (GitHub Pages)

Your repo already has a workflow that builds and deploys the game. You only need to turn on GitHub Pages and point it at that workflow.

## One-time setup on GitHub

1. Open your repo: **https://github.com/DataGod9000/2D_game_portfolio_website**
2. Go to **Settings** → **Pages** (left sidebar).
3. Under **Build and deployment**:
   - **Source:** choose **GitHub Actions** (not “Deploy from a branch”).
4. Save. You don’t need to create a workflow; it’s already in the repo.

## Deploy (automatic and manual)

- **Automatic:** Every push to `main` runs the workflow and deploys the latest build.
- **Manual:** **Actions** tab → **Deploy to GitHub Pages** → **Run workflow** → **Run workflow**.

## Your live URL

After the first successful run:

**https://datagod9000.github.io/2D_game_portfolio_website/**

(If your GitHub username is different, replace `datagod9000` with it.)

## If you use a different repo name

Edit `vite.config.js` and set `base` to your repo path:

```js
base: "/YOUR_REPO_NAME/",
```

Then redeploy (push to `main` or run the workflow again).

## Using a custom domain (optional)

In **Settings** → **Pages**, you can add a custom domain. Then set `base: "/"` in `vite.config.js` and redeploy.
