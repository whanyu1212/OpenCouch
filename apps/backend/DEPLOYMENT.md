# Private Cloud Run backend deployment

OpenCouch currently supports one backend worker in one active service instance.
The deployment workflow therefore disables the Cloud Run service, waits for
requests admitted under the configured timeout to finish, deploys a disabled
revision, moves all traffic to that revision, and only then starts one instance.
It does not support traffic splitting, tagged revisions, rolling overlap, or
unauthenticated public access.

The workflow is manual while the deployment architecture in issue #339 is being
validated. Run `.github/workflows/deploy-api.yml` from `main` only after the
required resources and variables below exist.

## One-time Google Cloud resources

Create the Artifact Registry repository if it is absent:

```bash
gcloud artifacts repositories create opencouch-api \
  --project fleet-anagram-244304 \
  --repository-format docker \
  --location us-central1 \
  --description "OpenCouch API images"
```

The current development deployment uses:

- project: `fleet-anagram-244304`
- region: `us-central1`
- Cloud Run service: `opencouch-api-dev`
- Cloud SQL connection: `fleet-anagram-244304:us-central1:free-trial-first-project`
- database URL secret: `opencouch-database-url`
- OpenAI API key secret: `OPENAI_API_KEY`

The runtime service account needs Cloud SQL Client and narrowly scoped Secret
Manager Secret Accessor access to both runtime secrets. The GitHub deployer
needs Artifact Registry write access, Cloud Run deployment access, permission
to act as the runtime service account, and Cloud Run Invoker access for the
authenticated smoke test. It also needs `iam.serviceAccounts.getOpenIdToken` on
the deployer service account to mint that smoke test's ID token; grant
`roles/iam.serviceAccountOpenIdTokenCreator` for this purpose. To preflight
runtime secrets before disabling the service, it needs the narrowly scoped
`secretmanager.versions.get` and `secretmanager.secrets.getIamPolicy`
permissions on both runtime secrets.

## GitHub Actions variables

Keep secret values in Google Secret Manager, not GitHub variables. Configure
these non-secret repository or `development` environment variables:

| Variable | Development value |
| --- | --- |
| `GCP_PROJECT_ID` | `fleet-anagram-244304` |
| `GCP_REGION` | `us-central1` |
| `AR_REPOSITORY` | `opencouch-api` |
| `CLOUD_RUN_SERVICE` | `opencouch-api-dev` |
| `CLOUD_SQL_INSTANCE` | `fleet-anagram-244304:us-central1:free-trial-first-project` |
| `DATABASE_URL_SECRET_NAME` | `opencouch-database-url` |
| `DATABASE_URL_SECRET_VERSION` | pinned enabled version, currently `1` |
| `OPENAI_API_KEY_SECRET_NAME` | `OPENAI_API_KEY` |
| `OPENAI_API_KEY_SECRET_VERSION` | pinned enabled version, currently `2` |
| `RUNTIME_SERVICE_ACCOUNT` | `opencouch-api-runtime@fleet-anagram-244304.iam.gserviceaccount.com` |
| `DEPLOYER_SERVICE_ACCOUNT` | `github-cloudrun-deployer@fleet-anagram-244304.iam.gserviceaccount.com` |
| `WORKLOAD_IDENTITY_PROVIDER` | configured GitHub Workload Identity provider resource name |
| `OPENCOUCH_CORS_ORIGINS` | optional exact staging web origin; never `*` |

`OPENCOUCH_MEMORY_DATABASE_URL` is not a GitHub variable. The workflow maps it
directly to the pinned `opencouch-database-url` Secret Manager version. Its
secret value is the application user's Cloud SQL DSN. The text-session store
reuses this shared URL unless an explicit text-session override is configured.

## Failure and recovery behavior

The deployment concurrency group never cancels an in-progress deployment. If
the workflow fails after scaling to zero, or if final verification fails after
activation, the service deliberately remains or returns to disabled rather than
restoring or serving an unverified revision. Inspect the failure, then either
rerun the workflow or explicitly route a known-good revision before setting
manual scaling back to one.

The final smoke check calls `/api/health` with an identity token. This proves
that the authenticated revision reached application startup; it is not a
substitute for the persistent text, memory, finalization, restart, voice, and
incognito checks tracked by issue #339.
