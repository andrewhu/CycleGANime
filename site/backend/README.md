## Backend setup

### Install dependencies
```
sudo apt install nginx redis-server wget git zip python3-venv vim tmux libsm6 libxext6 libxrender-dev -y
pip install -r requirements.txt
```

### Systemd configs
`redis.service`
```
[Unit]
Description=Advanced key-value store
After=network.target
Documentation=http://redis.io/documentation, man:redis-server(1)

[Service]
Type=forking
ExecStart=/usr/bin/redis-server /etc/redis/redis.conf
PIDFile=/run/redis/redis-server.pid
TimeoutStopSec=0
Restart=always
User=redis
Group=redis
RuntimeDirectory=redis
RuntimeDirectoryMode=2755

UMask=007
PrivateTmp=yes
LimitNOFILE=65535
PrivateDevices=yes
ProtectHome=yes
ReadOnlyDirectories=/
ReadWritePaths=-/var/lib/redis
ReadWritePaths=-/var/log/redis
ReadWritePaths=-/var/run/redis

NoNewPrivileges=true
CapabilityBoundingSet=CAP_SETGID CAP_SETUID CAP_SYS_RESOURCE
MemoryDenyWriteExecute=true
ProtectKernelModules=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictNamespaces=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX

# redis-server can write to its own config file when in cluster mode so we
# permit writing there by default. If you are not using this feature, it is
# recommended that you replace the following lines with "ProtectSystem=full".
ProtectSystem=true
ReadWriteDirectories=-/etc/redis

[Install]
WantedBy=multi-user.target
Alias=redis.service
```


`cycleganime.service`
```
[Unit]
Description=Gunicorn instance to serve cycleganime
After=network.target

[Service]
User=hupeechee
Group=www-data
WorkingDirectory=/var/www/CycleGANime/site/backend
Environment="PATH=/var/www/CycleGANime/site/backend/venv/bin"
ExecStart=/var/www/CycleGANime/site/backend/venv/bin/gunicorn --workers 3 --bind unix:server.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
```

`rq.service`
```
[Unit]
Description=RQ Worker
Requires=redis.service
After=redis.service

[Service]
Type=simple
WorkingDirectory=/var/www/CycleGANime/site/backend
Environment="PATH=/var/www/CycleGANime/site/backend/venv/bin"
ExecStart=/var/www/CycleGANime/site/backend/venv/bin/rq worker
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
PrivateTmp=true
Restart=always

[Install]
WantedBy=multi-user.target
```

