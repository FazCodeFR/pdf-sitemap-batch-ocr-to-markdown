import os
import time
import subprocess
import openstack

# Charger les variables d'environnement depuis le fichier openrc.sh
auth_url = os.getenv('OS_AUTH_URL')
project_name = os.getenv('OS_PROJECT_NAME')
username = os.getenv('OS_USERNAME')
password = os.getenv('OS_PASSWORD')
user_domain_name = os.getenv('OS_USER_DOMAIN_NAME')
project_domain_name = os.getenv('OS_PROJECT_DOMAIN_NAME')
region_name = os.getenv('OS_REGION_NAME')

# Connexion à OpenStack
conn = openstack.connect(
    auth_url=auth_url,
    project_name=project_name,
    username=username,
    password=password,
    region_name=region_name,
    user_domain_name=user_domain_name,
    project_domain_name=project_domain_name
)

# Obtenir la liste des instances
instances = list(conn.compute.servers())

# Vérifier s'il y a des instances
if instances:
    first_instance = instances[0]

    # Shelver l'instance
    conn.compute.shelve_server(first_instance.id)
    print(f"Instance '{first_instance.name}' avec l'ID '{first_instance.id}' a été shelvée.")

    # Forcer le offload immédiatement avec la commande CLI
    try:
        subprocess.run(
            ["openstack", "server", "shelve", "--offload", first_instance.id],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Commande shelve --offload exécutée avec succès pour '{first_instance.name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de shelve --offload : {e.stderr}")
        exit(1)

    # Vérifier l'état jusqu'à atteindre SHELVED_OFFLOADED
    while True:
        instance = conn.compute.get_server(first_instance.id)
        print(f"État actuel : {instance.status}")

        if instance.status == "SHELVED_OFFLOADED":
            print(f"Instance '{first_instance.name}' est maintenant en SHELVED_OFFLOADED.")
            break

        time.sleep(10)  # Vérifier toutes les 10 secondes

else:
    print("Aucune instance trouvée.")
