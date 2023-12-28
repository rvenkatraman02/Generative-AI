# Initializing Blob Storage Account

storage_account_name = BLOB_STORAGE_NAME
storage_account_access_key = BLOB_STORAGE_KEY
container_name = BLOB_STORAGE_CONTAINER
mount_point = MOUNT_LOCATION


# DBFS mount point directory

dbutils.fs.mount(
    source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
    mount_point = mount_point,
    extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
)

dbutils.fs.ls(mount_point)
